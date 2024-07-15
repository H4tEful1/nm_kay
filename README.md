To install and configure venv follow theese steps:
  1. Install development environment like PyCharm Community. It's free to use. 
    (https://www.jetbrains.com/pycharm/download/?section=windows) !!! Scroll down for community version, the big green button is for professional one !!!
    ![image](https://github.com/user-attachments/assets/9e546d2a-9e7a-4bef-93ab-b1e9abc462d8)
  2. Download project .zip from GitHub and unpack it to your working directory.
    ![image](https://github.com/user-attachments/assets/1d07b275-10cb-4db5-92ea-b94b84b9a7e9)
  3. Open it in Pycharm via File -> Open
     ![image](https://github.com/user-attachments/assets/7aee54d1-94a9-41a2-83f2-e33cb1cc1b6d)
  4. Configure venv for project:
       File -> Settings
       Project -> Python interpreter
       ![image](https://github.com/user-attachments/assets/1f33118c-7cec-47f2-bed6-60598d38abfa)
       Add a venv interpreter:
       ![image](https://github.com/user-attachments/assets/7c9740b6-47b9-4cf1-8249-b623d628c6e3)
       Activate venv so you can see a (venv) prefix before abs path in the terminal by command "venv\Scripts\activate" (for windows)
       ![image](https://github.com/user-attachments/assets/07d34f73-9b81-40a7-9316-377133404d83)
       Now install dependencies frome requirements.txt with command "pip install -r requirements.txt"
       Wait for installation and intrpreter upgrading
  5. Done

Didn't organized code from project yet so the only code it has is from "finetuning_fmri" colab, to try how it works run "kay_dload.py"
