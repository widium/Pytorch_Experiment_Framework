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

def append_info_in_log_file(
    logfile_path : Path,
    experiment_path : Path,
    experiment_name : str,
    last_train_accuracy : float,
    last_test_accuracy : float
)->None:
    
    log_info = f"\n****** {experiment_name.upper()} ******\n"
    log_info += f"- Path : [{experiment_path}]\n"
    log_info += f"- Train Accuracy : {last_train_accuracy:.2f}\n"
    log_info += f"- Test Accuracy : {last_test_accuracy:.2f}\n\n"

    with logfile_path.open("a") as file:
        file.write(log_info)
        print(f"[INFO] : Append {experiment_name} information in [{logfile_path}]")