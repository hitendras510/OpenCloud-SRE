import os

files_to_fix = [
    "README.md",
    "blog/blog.md",
    "notebooks/OpenCloud_SRE_Training.ipynb"
]

for file_path in files_to_fix:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Replace dhruv with hitendra
        fixed_content = content.replace("dhruv0431-sketch", "hitendras510")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)
            
        print(f"Fixed {file_path}")
    else:
        print(f"File not found: {file_path}")
