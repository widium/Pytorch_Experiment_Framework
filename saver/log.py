# *************************************************************************** #
#                                                                              #
#    log.py                                                                    #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/24 06:13:08 by Widium                                    #
#    Updated: 2023/04/24 06:13:08 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from pathlib import Path

# ================================================================================================ #

def check_log_file(root_path : str)->bool:
    
    root_path = Path(root_path)
    log_path = root_path / "log.txt"

    if log_path.is_file():
        print(f"[INFO] : [{log_path}] already initialized, append information inside")
        return (log_path)
    else :
        print(f"[INFO] : [{log_path}] doesn't exist, initialization...")
        log_path.touch()
        return (log_path)

# ================================================================================================ #