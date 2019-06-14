import csv
import re
import glob
import numpy as np
from keras.models import load_model
import face_recognition as fr

def main():
    model = load_model("face_rank_model.h5")
    files = glob.glob(r"data/*")
    for f in files:
        image = fr.load_image_file(f)
        encs = fr.face_encodings(image)
        if len(encs) != 1:
            print("Find %d faces in %s" % (len(encs), f))
            continue
        predicted = model.predict(np.array(encs))
        predicted = np.squeeze(predicted)
        list1=[re.split('\/',f)[-1],predicted]
        with open("result.csv", "a", newline='') as fff:
            csv.writer(fff).writerow(list1)
        print(type(predicted))
        print("%s: %.4f" % (f, predicted))

if __name__ == '__main__':
    main()