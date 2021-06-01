# JetsonMonitor
A simple Monitor based on Jetson nano 2GB.

You should firstly run FaceCollection.py to create your own 'FaceSet.dat' file, make sure that you have you roommates or family's images under a file folder named 'Faces', so that you can register their info. Also, take care of the content of codes, as you may need to change variables like the address of email, the absolute address of some file folder and so on.

Then, run Alert.py, this programme can monitor your room, if people registered come in, it can indentify their name. If someone unknown come in, this monitor will record their face encoding information and send you an email for each intruder.

I comment both in China and English for convenience of people from different region

I choose IMX219 camera module, you can find it here https://item.jd.com/10022506688217.html
