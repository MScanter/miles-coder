import os



def read_file(path):
    with open(path,"r") as f:
        return f.read()
    

def write_file(path,content):
    with open(path,"w") as w:
        w.write(content)


def list_files(directory):
    return os.listdir(directory)

TOOLS_MAP = {
      "read_file": read_file,
      "write_file": write_file,
      "list_files": list_files,
  }


TOOLS_SCHEMA = [
      {"type": "function", "function": {"name": "read_file", "description":
  "读取文件内容", "parameters": {"type": "object", "properties": {"path": {"type":
  "string"}}, "required": ["path"]}}},
      {"type": "function", "function": {"name": "write_file", "description":
  "写入文件", "parameters": {"type": "object", "properties": {"path": {"type":
  "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
      {"type": "function", "function": {"name": "list_files", "description":
  "列出目录文件", "parameters": {"type": "object", "properties": {"directory":
  {"type": "string"}}, "required": ["directory"]}}},
  ]

