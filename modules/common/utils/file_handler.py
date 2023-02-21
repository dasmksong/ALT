import glob
import os
from typing import Dict

if os.name == "nt":
    import win32api
    import win32con


def file_is_hidden(dir: str, p: str) -> bool:
    """파일이 숨김파일인지 확인, 숨김 파일이면 True, 아니면 False 리턴

    Args:
        p (str): 파일 경로

    Returns:
        _type_ : 숨김파일 여부
    """
    if os.name == "nt":
        attribute = win32api.GetFileAttributes(os.path.join(dir, p))
        return attribute & (
            win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM
        )
    else:
        return p.startswith(".")  # linux-osx


def get_file_list(
    directory_path: str,
    hidden_file: bool = False,
    files_only: bool = True,
    inc_subdir: bool = False,
) -> Dict:
    """폴더 안의 파일과 폴더들의 path list와 name list를 dict 형태로 반환

    Args:
        directory_path (str): 파일 경로
        hidden_file (bool) : 숨김 파일 혹은 폴더 적용 여부
        files_only (bool) : 폴더 제외 여부
        inc_subdir (bool) : 내부 폴더의 파일까지 적용 여부

    Returns:
        dict : path list와 name list
    """

    def is_file(file_path: str):
        """파일이 존재하는지 확인, file_path 가 비어있거나 file_path에 파일이 존재할 경우 True, file_path에 파일이 존재할 경우 False 리턴

        Args:
            file_path (str): 파일 경로

        Returns:
            bool : 파일 존재 여부
        """
        if files_only:
            return os.path.isfile(file_path)
        else:
            return True

    names = []
    full_paths = []
    file_list = []
    if inc_subdir:
        for root, dirs, files in os.walk(directory_path):
            if len(files) < 1 and is_file(root):
                names.append(os.path.basename(root))
                full_paths.append(root)

            for file in files:
                full_path = os.path.join(root, file)
                if hidden_file:
                    if is_file(full_path):
                        names.append(file)
                        full_paths.append(full_path)
                else:
                    if is_file(full_path) and not file_is_hidden(root, file):
                        names.append(file)
                        full_paths.append(full_path)
    else:
        if hidden_file:
            file_list = [
                (os.path.join(directory_path, f), f)
                for f in os.listdir(directory_path)
                if is_file(os.path.join(directory_path, f))
            ]

        else:
            file_list = [
                (os.path.join(directory_path, f), f)
                for f in os.listdir(directory_path)
                if (
                    (not file_is_hidden(directory_path, f))
                    and (is_file(os.path.join(directory_path, f)))
                )
            ]
        full_paths = [f[0] for f in file_list]
        names = [f[1] for f in file_list]

    return {"fullpath": full_paths, "name": names}


def chk_and_make_dir(directory_path: str) -> None:
    """해당 폴더가 없으면 만들고, 있으면 pass

    Args:
        directory_path (str): 폴더 경로

    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_last_path(dir: str, type: str = "dir", by: str = "name") -> str:
    """배열 기준에 따라 경로 안 폴더 혹은 파일의 경로 반환

    Args:
        dir (str): 파일 경로
        type (str): 찾으려는 type
        by (str): 배열 기준

    Returns:
        str : 해당 폴더 혹은 파일의 경로
    """
    if type == "dir":
        dir_list_class = list(filter(os.path.isdir, glob.glob(dir + "/*")))
    elif type == "file":
        dir_list_class = list(filter(os.path.isfile, glob.glob(dir + "/*")))
    elif type == "all":
        dir_list_class = glob.glob(dir + "/*")
    else:
        dir_list_class = glob.glob(dir + "/*" + type)

    last_path = ""

    if len(dir_list_class) > 0:
        if by == "mtime":
            sort_by = os.path.getmtime
        elif by == "ctime":
            sort_by = os.path.getctime
        elif by == "atime":
            sort_by = os.path.getatime
        elif by == "name":
            return max(dir_list_class)

        last_path = max(dir_list_class, key=sort_by)
    return last_path
