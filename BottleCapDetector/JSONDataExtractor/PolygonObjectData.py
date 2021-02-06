class PolygonObjectData:
    def __init__(self, label, points):
        self.Label = label
        self.Points = points
    
    def __str__(self):
        returnString = 'Label: ' + self.Label + ', \t'
        returnString += 'Points: {'
        for point in self.Points:
            returnString += '[' + str(point[0]) + ',' + str(point[1]) + ']'
        returnString += '}'
        return returnString
