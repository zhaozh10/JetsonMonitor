import face_recognition 
import cv2
import os
import platform
import pickle
import sys
#存储知道人名列表
known_names=[] 
#存储知道的特征值
known_encodings=[]


#将目前已保存在列表中的人脸信息保存到.dat文件中（‘w'代表如果known_faces.dat文件中已有内容，则会全部覆盖重写该文件，若要实现“增加”的功能，则应该是'a'）
def save_known_faces():
	#将know_faces.dat文件读入为face_data_file，并设置为“写二进制文件"模式
    with open("FaceSet.dat", "wb") as face_data_file:
        face_data = [known_encodings, known_names]
	# pickle.dump(obj,file,[,protocol])序列化对象，并将obj保存到file中,file必须是可写入的
        pickle.dump(face_data, face_data_file)
        #print("Known faces backed up to disk with name {}.".format(known_face_names[-1]))



def face(path):
    # 'path'是存放人脸图像的路径
    # 'path' is the address of file folder which contains different people's face image.
    for image_name in os.listdir(path):
        known_names.append(image_name.split(".")[0])
        # 加载图片
        # load images
        load_image = face_recognition.load_image_file(os.path.join(path,image_name)) 
        Face_locations=face_recognition.face_locations(load_image)
        # 不允许一张图片中有多个人脸，这会导致known_names和known_encodings对不上
        # We don't allow more than one face in a image, as we hope each the length of known_names can match known_encodings
        if len(Face_locations)!=1:
            raise Exception("Multiple faces in one picture!")
        print("We find {} face(s)".format(len(Face_locations)))
        # 获得128维特征值
        # get 128-dimension feature vector
        image_face_encodings= face_recognition.face_encodings(load_image,Face_locations,model='large') 
        # 这个for循环是痛点，不加这个的话，会在Alert程序里出现莫名其妙到维度匹配错误
        # We need this for loop to make Alert.py work properly, otherwise the dimension can mismatch 
        for face_encoding in image_face_encodings:
            known_encodings.append(face_encoding)
        print("length of name: ",len(known_names)-1)
        print("length of encodings: ",len(known_encodings)-1)
        print(known_names[-1])
if __name__=='__main__':
    face("YOU_IMAGE_FOLDER_ADDRESS") 
    print(known_names)
    save_known_faces()
