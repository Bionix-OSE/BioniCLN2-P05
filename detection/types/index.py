class MangoImage:
    def __init__(self, file_path: str, metadata: dict = None):
        self.file_path = file_path
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"MangoImage(file_path={self.file_path}, metadata={self.metadata})"

    def get_metadata(self):
        return self.metadata

    def set_metadata(self, key: str, value):
        self.metadata[key] = value

    def get_file_path(self):
        return self.file_path