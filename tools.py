import os
from pydantic import BaseModel, Field


class ReadFileArgs(BaseModel):
    path: str = Field(description="文件的完整路径")

class WriteFileArgs(BaseModel):
    path: str = Field(description="文件的完整路径")
    content: str = Field(description="要写入的内容")

class ListFilesArgs(BaseModel):
    directory: str = Field(description="目录的完整路径")

class EditFileArgs(BaseModel):
    path: str = Field(description="文件的完整路径")
    old_content: str = Field(description="要替换的原内容")
    new_content: str = Field(description="要替换的新内容")


def read_file(path):
    with open(path,"r") as f:
        return f.read()
    

def write_file(path,content):
    with open(path,"w") as w:
        w.write(content)
    return f"已写入{len(content)}个字符到{path}"


def list_files(directory):
    return os.listdir(directory)


def edit_file(path,old_content,new_content):
    with open(path,"r") as f:
        content = f.read()
    if old_content not in content:
        return f"错误: 未找到要替换的内容"
    content = content.replace(old_content,new_content,1)
    with open(path,"w") as w:
        w.write(content)
    return f"已修改{path}"

TOOLS_MAP = {
      "read_file": read_file,
      "write_file": write_file,
      "list_files": list_files,
      "edit_file": edit_file,
  }


def make_tool(name: str, description: str, args_model: type[BaseModel]):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": args_model.model_json_schema()
        }
    }


TOOLS_SCHEMA = [
    make_tool("read_file", "读取文件内容", ReadFileArgs),
    make_tool("write_file", "写入文件", WriteFileArgs),
    make_tool("list_files", "列出目录文件", ListFilesArgs),
    make_tool("edit_file", "编辑文件内容", EditFileArgs),
]


# 旧的手写 TOOLS_SCHEMA（已用 make_tool 替代）
# TOOLS_SCHEMA = [
#     {
#         "type": "function",
#         "function": {
#             "name": "read_file",
#             "description": "读取指定路径的文件内容",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "path": {"type": "string", "description": "文件的完整路径"}
#                 },
#                 "required": ["path"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "write_file",
#             "description": "将内容写入指定路径的文件",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "path": {"type": "string", "description": "文件的完整路径"},
#                     "content": {"type": "string", "description": "要写入的内容"}
#                 },
#                 "required": ["path", "content"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "list_files",
#             "description": "列出指定目录下的所有文件和文件夹",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "directory": {"type": "string", "description": "目录的完整路径"}
#                 },
#                 "required": ["directory"]
#             }
#         }
#     },
# ]