import cv2
import mediapipe as mp
import math as m


camera = cv2.VideoCapture(0)
draw = mp.solutions.drawing_utils
hand = mp.solutions.hands

with hand.Hands(static_image_mode=True) as hands:
    while (camera.isOpened()):
        control, image = camera.read()
        final = hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

        y,x,_=image.shape

        if final.multi_hand_landmarks != None:
            for hand_landmark in final.multi_hand_landmarks:
                for cordinate in hand.HandLandmark:
                    cordinate1 = hand_landmark.landmark[4]
                    cordinate2 = hand_landmark.landmark[8]
                    cv2.circle(image,(int(cordinate1.x * x), int(cordinate1.y * y)), 4, (255, 0, 0), 4)
                    cv2.circle(image,(int(cordinate2.x * x), int(cordinate2.y * y)), 4, (255, 0, 0), 4)
                    cv2.line(image,(int(cordinate1.x * x), int(cordinate1.y * y)),(int(cordinate2.x * x), int(cordinate2.y * y)),(255,0,0),3)
                    distance = int(m.sqrt(m.pow((cordinate2.x-cordinate1.x)*x,2)+m.pow((cordinate2.y-cordinate1.y)*y,2)))
                    cv2.putText(image,"Distance: "+f"{int(distance)}",(50,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        cv2.imshow("image",image)

        cv2.waitKey(250)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()