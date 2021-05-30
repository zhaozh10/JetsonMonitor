import face_recognition
import cv2
import yagmail
from datetime import datetime, timedelta
import jetson.utils
import numpy as np
import platform
import numpy as np
import pickle
import sys
sys.path.append("CURRENT PATH")
from Repo import NameGenerator

# 这两个列表是全局的，其中known_face_name是各元素为字典的列表
# a list saving each register's face encoding info
known_face_encodings = []
# a list saving each register's name
known_face_name = []
# a list saving unknown visitor's face encoding info
unknown_visitor=[]

def SendEmail(remark,address):
    # 用来发信的邮箱/该邮箱提供的授权码/服务器host
    # email used for sending/This type of email's permission code/corresponding host
    yag=yagmail.SMTP(user="YOU_EMAIL_ACCOUNT@163.com",password="YOU_PASSWORD",host='smtp.163.com')
    # send email to TARGET_EMAIL/email subject is 'Intruders'/
    # variable 'remark' is the email's content/variable 'address' is image's storage address
    yag.send('TARGET_EMAIL@qq.com','Intruders',remark,address)

# 将目前已保存在.dat文件中的人脸信息加载为列表
# load saved info from 'FaceSet.dat', which is the output of programme 'FaceCollection'
def load_known_faces():
    global known_face_encodings, known_face_name

    try:
        with open("FaceSet.dat", "rb") as face_data_file:
            known_face_encodings, known_face_name = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass



# 在已有到面孔信息中查找，判断目前检测到的这个人是否来过了
# See if this is a face we already have in our face list
def lookup_known_face(face_encoding):
    
    name = None

    # If our known face list is empty, just return 'None'
    if len(known_face_encodings) == 0:
        return name

    # 计算目前检测到的人脸和已保存的各个人脸到相似度，并输出最像的那个人脸编号
    # 返回一个列表face_distance里面包含了当前人脸与各个已保存人脸的特征空间距离，范围是0-1，距离为0时，说明是同一个面孔
    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    #print(known_face_name[best_match_index]," with ",best_match_index)

    # 如果最短距离小于threshhold,则认为当前人脸已出现过，并更新该人脸对应到name
    if face_distances[best_match_index] < 0.5:
        # If we have a match, look up the name we've saved for it 
	    # 在Python中直接赋值一个列表的话，这两个列表是完完全全地浅拷贝，共用一个地址，修改其中一个会影响另一个
        name = known_face_name[best_match_index]
        
    return name

#在已有到面孔信息中查找，判断目前检测到的这个人是否来过了
# we only want to save one image for each unknown visitor, 
# so we need this function to judge whether this unknown visitor's image has been saved,
# otherwise you email may receive hundreds of emails concentrating on the same visitor  
def lookup_unknown_face(face_encoding):
    
    flag = False
    
    # If our known face list is empty, just return nothing
    if len(unknown_visitor) == 0:
        return flag

    # 计算目前检测到的人脸和已保存的各个人脸到相似度，并输出最像的那个人脸编号
    # 返回一个列表face_distance里面包含了当前人脸与各个已保存人脸的特征空间距离，范围是0-1，距离为0时，说明是同一个面孔
    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    face_distances = face_recognition.face_distance(unknown_visitor, face_encoding)
    best_match_index = np.argmin(face_distances)
    # 如果最短距离小于threshhold,则认为当前人脸已出现过，并更新该人脸对应到name
    if face_distances[best_match_index] < 0.6:
        # 我们辨认出这个来访者在前面的帧中出现过，但我们无须返回这个未知来访者的信息，只需要返回True就行了
        # we no longer need to return this unknown visitor's info,if we have a match, just return 'True'
        flag=True
    return flag

def Record_live():
    dispW=1280
    dispH=720
    # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
    # We choose jetson.utils.gstCamera as it can accelerate video I/O stream
    camera=jetson.utils.gstCamera(dispW,dispH,"0")
    nVisitor=0

    while True:
        # Grab a single frame of video
        img,_,_=camera.CaptureRGBA(zeroCopy=True)
        # 开启同步
        # Synchronize
        jetson.utils.cudaDeviceSynchronize()
        # cuda转Numpy，从而进行后续的OpenCv操作
        # Convert from cuda to Numpy for convenience of coming OpenCv operation
        frame=jetson.utils.cudaToNumpy(img,1280,720,4)
        frame=cv2.cvtColor(frame.astype(np.uint8),cv2.COLOR_RGB2BGR)
        
        # 将当前帧缩放为1/4的尺寸，以便于人脸识别
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # 通道转换
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

	    # 在这里就获得了当前人脸在当前帧中的位置和encodings
        # Find all the face locations and face encodings in the current frame of video
	    # face_locations返回的是一个元组，其中各元素是(top, right, bottom, left)
        # face_locations return a tuple seemed like (top, right, bottom, left)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations,model='large')
        # 在当前帧，对每张侦测到的人脸都进行比较，判断是否是已注册用户
        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # 调用lookup_known_face函数来判断该人脸是否是已注册的
            # See if this face is in our list of known faces.
            name = lookup_known_face(face_encoding)

            
	        # 如果当前这个人脸是已被注册的，则返回的name非空，则face_label等于此人的姓名 
	        # 如果是未注册的人脸，则首先判断是否是前面的帧中拍到的来访者，如果是，则仅仅为其打上Unknown的标签
            # 如果在前面帧中未被拍到，则记录到unknown_visitor中，拍照，发邮件
            # If this face has been registered, face_label=name.
            # If this face hasn't been registered and has been recorded in list, we just label it 'Unknown'
            # If this face hasn't been registered and shows up the first time, we record/shot and send email 
            if name is not None:
                face_label=name
            elif lookup_unknown_face(face_encoding):
                face_label = "Unknown"
            else:
                face_label = "Unknown"
                # 获取当前帧的图片
                # shot
                top, right, bottom, left = face_location
                shot_image=small_frame
                # 将未知来访者的人脸编码添加到list中
                # append face_encoding
                unknown_visitor.append(face_encoding)
                nVisitor=nVisitor+1
                # 在本地储存拍摄的图片，并发送邮件
                # save the images and send email
                order="No.{}-".format(nVisitor)
                pre=datetime.now().strftime('Day%d_%H-%M-%S')
                remark=order+pre
                address="Intruder/"+order+pre+".jpg"
                cv2.imwrite(address,shot_image)
                SendEmail(remark,address)

            face_labels.append(face_label)
        # 在人脸位置画一个框，框上有对应的label
        # Draw a box around each face and label each face
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	        # 输入参数依次为：image,text,字符串左上角的坐标,字体，字号大小，字体颜色，字体宽度
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)



        # Display the final frame of video with boxes drawn around each detected fames
        cv2.imshow('Video', frame.astype(np.uint8))


        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break
    #video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_known_faces()
    print(known_face_name)
    Record_live()
