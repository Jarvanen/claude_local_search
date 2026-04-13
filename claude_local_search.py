import os
import re
import glob as pyglob
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import tiktoken

# 初始化 FastAPI 应用
app = FastAPI(
    title="Local Codebase Agent Tools",
    description="用于 Agent 检索和读取本地代码库的工具集 (FileRead, Glob, Grep)",
    version="1.0.0"
)

# ==========================================
# 安全设置：定义 Agent 允许访问的根目录
# 请将其修改为你要测试的代码库绝对路径
# ==========================================
WORKSPACE_DIR = Path("/Users/turing/Desktop/langchain").resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
USER_PROFILE_DIR = WORKSPACE_DIR / "agent_data" / "profiles"
USER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHARS_PER_READ = 20000 #读取文件内容最大长度
MAX_CONTEXT_TOKENS = 131072  # 超过此值触发压缩
ENCODING_NAME = "cl100k_base" # Claude/GPT-4 使用的编码


def get_safe_path(target_path: str) -> Path:
    if not target_path: return WORKSPACE_DIR
    resolved = (WORKSPACE_DIR / target_path).resolve()
    if not str(resolved).startswith(str(WORKSPACE_DIR)):
        raise HTTPException(status_code=403, detail="Access Denied")
    return resolved

# ==========================================
# 功能 1: 用户画像管理 (User Profile)
# ==========================================

class ProfileUpdateRequest(BaseModel):
    user_id: str
    description: str  # 模型生成的画像描述


@app.post("/api/user/profile", tags=["User Management"])
def save_user_profile(req: ProfileUpdateRequest):
    """保存或更新用户画像描述"""
    profile_path = USER_PROFILE_DIR / f"{req.user_id}.json"
    try:
        data = {"description": req.description}
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"status": "success", "message": f"Profile saved for {req.user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/profile/{user_id}", tags=["User Management"])
def get_user_profile(user_id: str):
    """读取用户画像，用于放入 System Prompt"""
    profile_path = USER_PROFILE_DIR / f"{user_id}.json"
    if not profile_path.exists():
        return {"description": ""}
    with open(profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==========================================
# 功能 2: Token 监控与内容压缩 (Compression)
# ==========================================

def count_tokens(text: str) -> int:
    """计算文本的 Token 数"""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(text))


class CompressionRequest(BaseModel):
    content: str
    max_tokens: Optional[int] = MAX_CONTEXT_TOKENS


@app.post("/api/utils/compress", tags=["Utils"])
def compress_content(req: CompressionRequest):
    """
    识别内容是否超长并建议压缩。
    注意：此工具返回压缩建议，实际总结动作通常由 Agent 调用 LLM 完成。
    """
    tokens = count_tokens(req.content)

    if tokens <= req.max_tokens:
        return {"needs_compression": False, "current_tokens": tokens}

    # 策略：如果超长，返回头部和尾部的“采样”，并提示模型进行总结
    # 就像 Claude Code 的 Compaction 过程
    lines = req.content.splitlines()
    head = "\n".join(lines[:20])
    tail = "\n".join(lines[-20:])

    compressed_hint = (
        f"[SYSTEM: 内容过长 ({tokens} tokens)，已自动采样]\n"
        f"--- 头部内容 ---\n{head}\n"
        f"...\n[中间 {len(lines) - 40} 行已隐藏]\n"
        f"...\n--- 尾部内容 ---\n{tail}\n"
        f"[请模型根据上述采样总结核心信息，以减少上下文占用]"
    )

    return {
        "needs_compression": True,
        "current_tokens": tokens,
        "suggested_content": compressed_hint
    }


# ==========================================
# 1. FileReadTool (读取文件内容)
# ==========================================
class FileReadRequest(BaseModel):
    file_path: str = Field(..., description="要读取的绝对或相对文件路径")
    offset: Optional[int] = Field(0, description="读取的起始行号偏移量 (从0开始)")
    limit: Optional[int] = Field(1000, description="最多读取的行数")


@app.post("/api/tools/read_file", summary="读取文件 (FileRead)", tags=["Tools"])
def read_file(req: FileReadRequest):
    safe_path = get_safe_path(req.file_path)

    if not safe_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    try:
        with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        start = req.offset or 0
        limit = req.limit or 1000
        end = min(start + limit, total_lines)

        sliced_lines = all_lines[start:end]

        formatted_content = ""
        current_chars = 0
        actual_end_line = start
        is_truncated_by_chars = False

        for i, line in enumerate(sliced_lines):
            line_num = start + i + 1
            line_text = f"{line_num}\t{line}"

            # 检查是否超过单次 Token/字符限制
            if current_chars + len(line_text) > MAX_CHARS_PER_READ:
                is_truncated_by_chars = True
                break

            formatted_content += line_text
            current_chars += len(line_text)
            actual_end_line = line_num

        # 构建返回结果
        response = {
            "file_path": str(safe_path.relative_to(WORKSPACE_DIR)),
            "total_lines": total_lines,
            "read_range": f"lines {start + 1}-{actual_end_line}",
            "is_truncated": is_truncated_by_chars or (end < total_lines),
            "content": formatted_content
        }

        # 核心压缩技巧：如果被截断，在结果中添加“导航提示”
        if response["is_truncated"]:
            remaining = total_lines - actual_end_line
            truncation_msg = f"\n\n[NOTE: Content truncated. {remaining} lines remaining. " \
                             f"To read more, call read_file with offset={actual_end_line}]"
            response["content"] += truncation_msg

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 2. GlobTool (按文件名模式查找)
# ==========================================
class GlobRequest(BaseModel):
    pattern: str = Field(..., description="Glob 匹配模式，例如 '**/*.ts' 或 'src/**/*.py'")
    path: Optional[str] = Field(None, description="搜索起始目录。不填则默认在当前工作区搜索")


@app.post("/api/tools/glob", summary="模式匹配查找文件 (Glob)", tags=["Tools"])
def glob_search(req: GlobRequest):
    """
    基于 glob 模式在代码库中查找文件路径。
    """
    search_root = get_safe_path(req.path)
    if not search_root.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    try:
        # 使用 Python 内置的 glob 模块进行递归搜索
        matches = pyglob.glob(req.pattern, root_dir=str(search_root), recursive=True)

        # 过滤掉目录，只保留文件，并转换为相对路径
        file_matches = []
        for match in matches:
            full_path = search_root / match
            if full_path.is_file():
                file_matches.append(str(full_path.relative_to(WORKSPACE_DIR)))

        return {
            "num_files": len(file_matches),
            "filenames": file_matches[:200]  # 限制最大返回数量防止 token 溢出
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 3. GrepTool (正则文本搜索)
# ==========================================
class GrepRequest(BaseModel):
    pattern: str = Field(..., description="要搜索的正则表达式，例如 'function\\s+\\w+'")
    path: Optional[str] = Field(None, description="搜索的起始目录或文件")
    include_glob: Optional[str] = Field(None, description="要包含的文件过滤模式，例如 '*.py'")


@app.post("/api/tools/grep", summary="全文正则搜索 (Grep)", tags=["Tools"])
def grep_search(req: GrepRequest):
    """
    纯 Python 实现的类似 ripgrep 的功能，在文件中搜索指定的正则表达式。
    """
    search_root = get_safe_path(req.path)

    if not search_root.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    try:
        regex = re.compile(req.pattern)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {str(e)}")

    results = []
    MAX_MATCHES = 100

    def search_in_file(filepath: Path):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_idx, line in enumerate(f):
                    if regex.search(line):
                        # 压缩处理：只保留前 N 个字符，防止单行过长（如压缩后的 JS 文件）
                        clean_line = line.strip()
                        if len(clean_line) > 500:
                            clean_line = clean_line[:500] + "... [line truncated]"

                        results.append({
                            "file": str(filepath.relative_to(WORKSPACE_DIR)),
                            "line": line_idx + 1,
                            "content": clean_line
                        })
                        if len(results) >= MAX_MATCHES:
                            return
        except Exception:
            pass

    # 如果目标是单文件
    if search_root.is_file():
        search_in_file(search_root)
    # 如果目标是目录
    else:
        # 如果指定了 include_glob，则使用 rglob 匹配；否则遍历所有文件
        file_iterator = search_root.rglob(req.include_glob) if req.include_glob else search_root.rglob("*")

        for p in file_iterator:
            if p.is_file():
                # 简单过滤常见忽略目录
                if any(part in p.parts for part in ['.git', 'node_modules', '__pycache__', 'venv']):
                    continue
                search_in_file(p)

    return {
        "matches_count": len(results),
        "total_found_hint": "Displaying first 100 matches. Use a more specific regex if needed.",
        "results": results
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9002)