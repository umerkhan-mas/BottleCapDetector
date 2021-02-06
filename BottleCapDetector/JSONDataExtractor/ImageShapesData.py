class ImageShapesData:
    def __init__(self, imageName, shapes):
        self.ImageName = imageName
        self.Shapes = shapes

    def __str__(self):
        returnString  = 'Image Name : ' + self.ImageName + '\n'

        for i in range(len(self.Shapes)):
            returnString += 'Shape ' + str(i) + ' ==> ' + str(self.Shapes[i]) + '\n'

        return returnString