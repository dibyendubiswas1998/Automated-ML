from datetime import datetime

class App_Logger:
    """
        This class is responsible for log all the details with particular file.
    """
    def __init__(self):
        pass

    def log(self, file_object, log_message):
        """
            Method Name: log
            Description: This method log the details

            Output: log the details.
            On Failure: Raise Exception

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + '\n')  # log the details with time stamp

if __name__ == '__main__':
    pass


