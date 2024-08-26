import glob
import json
import random
import trimesh
import numpy as np
import traceback
import csv
import torch

LANDMARKS_CLASS = ["Mesial", "Distal", "Cusp", "InnerPoint", "OuterPoint", "FacialPoint"]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class LandmarkDet:  # LandmarkDetectionAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """

        #self.model = load_model()
        #sef.device = "cuda"

        pass

    @staticmethod
    def load_input(input_dir):
        """
        Read from /input/
        """
        # iterate over files in input_dir
        inputs = glob.glob(f'{input_dir}/*.obj')
        print("scan to process:", inputs)
        return inputs

    @staticmethod
    def write_output(predictions_list, output_dir="/output"):
        """
        #TODO add document here
        """
        # with open(f'{output_dir}/predictions.json', 'w') as fp:
        #     json.dump(predictions_list, fp, cls=NpEncoder)
        # Writing the list of dictionaries to a CSV file
        with open(f'{output_dir}/predictions.csv', mode='w', newline='') as file:
            fieldnames = ['key', 'coord_x', 'coord_y', 'coord_z', 'class', 'score']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            # Write the data
            for sublist in predictions_list:
                for landmark in sublist:
                    # Flatten the coord list for CSV
                    coord_x, coord_y, coord_z = landmark['coord']
                    row = {
                        'key': landmark['key'],
                        'coord_x': coord_x,
                        'coord_y': coord_y,
                        'coord_z': coord_z,
                        'class': landmark['class'],
                        'score': landmark['score']
                    }
                    writer.writerow(row)
        return

    def predict(self, inputs):
        """
        Your algorithm goes here
        """
        landmarks_predicted = []
        for scan_path in inputs:
            scan_name = scan_path.split('/')[-1].split('.obj')[0]

            print(f"loading scan : {scan_name}")
            # read input 3D scan .obj
            try:
                # you can use trimesh or other any loader we keep the same order
                mesh = trimesh.load(scan_path, process=False)
            except Exception as e:
                print(str(e))
                print(traceback.format_exc())
                raise
            # preprocessing if needed
            # prep_data = preprocess_function(mesh)
            # inference data here
            # landmarks = self.model(mesh)

            # extract number of vertices from mesh
            nb_vertices = mesh.vertices.shape[0]
            landmarks_scan = [
                {"key": scan_name,  # ensure that the key is the scan name
                 "coord": mesh.vertices[0], #mesh.vertices[random.randint(0, nb_vertices)],  # xyz coordinate
                 "class": LANDMARKS_CLASS[0],#random.choice(LANDMARKS_CLASS),  # ensure that the class should be in LANDMARKS_CLASS list
                 "score": 1#random.random()  # a prediction likelihood between 0 and 1
                 },
                {"key": scan_name,
                 "coord": mesh.vertices[1],  # mesh.vertices[random.randint(0, nb_vertices)],
                 "class": LANDMARKS_CLASS[1],  # random.choice(LANDMARKS_CLASS),
                 "score": 1  # random.random()
                 },
            # ....
            ]

            landmarks_predicted.append(landmarks_scan)

        return landmarks_predicted

    def process(self):
        """
        Read input from /input, process with your algorithm and write to /output
        """
        input = self.load_input(input_dir='/input')
        landmarks_predicted = self.predict(input)
        self.write_output(predictions_list=landmarks_predicted, output_dir='/output')


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available!")
    else:
        print("CUDA is not available.")
    LandmarkDet().process()
