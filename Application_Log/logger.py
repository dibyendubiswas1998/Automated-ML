from datetime import datetime

class App_Logger:
    def __init__(self):
        pass

    def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + '\n')

if __name__ == '__main__':
    logger = App_Logger()

    file = open("../Executions_Logs/Training_Logs/Genereal_Logs.txt", 'a+')
    logger.log(file, "Okay It's fine")
    file.close()

