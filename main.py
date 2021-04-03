import cv2

cap = cv2.VideoCapture(1)
i=0
while(True):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh_binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    cv2.imwrite('dataset/five/five'+str(i)+'.jpg', thresh_binary)
    cv2.imshow('image', thresh_binary)
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

