

class ConsoleOutput:
    def GetString(self, dictionary):
        output_string = ''
        for key, value in dictionary.items():
            output_string += "The quantity for label '{label}' is:{length}.\n".format(label=key, length=len(value))
        return output_string

    def Print(self, dictionary):
        output_string = self.GetString(dictionary)
        print(output_string)
        