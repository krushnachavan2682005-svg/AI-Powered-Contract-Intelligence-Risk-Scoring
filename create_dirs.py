import os

def create_dirs():
    dirs = [
        "src/inference",
        "src/api",
        "tests/integration"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        init_file = os.path.join(d, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                pass

if __name__ == "__main__":
    create_dirs()
    print("Created dirs")
