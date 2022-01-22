import math
from PIL import Image
import numpy
import pandas


class knn_model:
    def __init__(self, step):
        self.data = []
        self.data_labels = []
        self.step = step

    def get_data(self):
        print("Fetching training data...")

        data = pandas.read_csv("image_data_condensed.csv")
        data.values.tolist()
        
        for i in range(1, len(data["class"]), self.step):
            self.data_labels.append(int(data.loc[i].values.tolist()[-1]))
        
        for i in range(1, len(data), self.step):
            row = data.loc[i].values.tolist()
            self.data.append(row[2:-1])

        print("Fetched training data")

    def get_nearest(self, point):
        print("Finding nearest neighbours...", "\n")

        diffs = []
        for j, i in enumerate(self.data):
            val = 0
            for x, y in enumerate(point):
                val += (y - i[x])**2

            diffs.append([math.sqrt(val), self.data_labels[j]])

        diffs.sort()

        predictions = []
        for k in range(1, 18, 2):
            nums = [0 for _ in range(10)]
            for i in range(k):
                nums[diffs[i][1]] += 1

            highest = [0, 0]
            for i, x in enumerate(nums):
                if x > highest[0]:
                    highest = [x, i]

            predictions.append(highest[1])
            print("Predicted:", highest[1], "when k =", k)

        vals = [[predictions.count(i), i] for i in range(10)]
        vals.sort(reverse=True)

        print("Final prediction:", vals[0][1])


def get_pixel_vals(file_name):
    img = Image.open(file_name).convert("L")
    img_array = numpy.array(img)

    vals = []
    for i in img_array:
        for x in i:
            vals.append(x)

    return vals


model = knn_model(1)
model.get_data()
model.get_nearest(get_pixel_vals("handwriting_image.bmp"))