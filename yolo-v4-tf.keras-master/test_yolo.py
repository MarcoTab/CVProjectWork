from models import Yolov4
import os

model = Yolov4(weight_path='../weights/yolov4.weights', 
               class_name_path='class_names/coco_classes.txt')

outputdir = "../yolo_outs/"
for i in range(140, 225):
    these_preds = model.predict(f'../Images_61/0{i}.png', plot_img=False)

    os.makedirs(os.path.join(outputdir, f"0{i}"), exist_ok=True)

    # print(list(these_preds.iloc[these_preds["class_name"] == "car"]))
    # break

    for j, car in these_preds.loc[these_preds["class_name"] == "car"].iterrows():
        print(car)
        with open(os.path.join(outputdir, f"0{i}", f"{j}.txt"), 'w') as fp:
            fp.write(str(car['x1'])+'\n')
            fp.write(str(car['y1'])+'\n')
            fp.write(str(car['x2'])+'\n')
            fp.write(str(car['y2'])+'\n')
    

# predictions = model.predict()

# print(predictions)