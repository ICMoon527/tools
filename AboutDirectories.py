import os

class AboutDirectories():
    def __init__(self, path) -> None:
        self.path = path

    def deleteSpace(self, path):
        dirs = os.listdir(path)
        for dir in dirs:
            if '\xa0' in dir:
                new_name = dir.replace('\xa0', '')
                os.rename(os.path.join(path, dir), os.path.join(path, new_name))

if __name__ == '__main__':
    object = AboutDirectories('G:\\')
    object.deleteSpace('G:\\ColorectalCancer')