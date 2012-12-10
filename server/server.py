from bottle import route, run, request, static_file, redirect
import random
import Image
import os

@route('/')
@route('/index')
def index():
    resp = """<html><head><title>iHOG Demo</title></head>
<body style="background-color:#EFEFEF;font-family:Arial;">
<div style="margin : 20px auto; padding:20px; background-color:#fff;width:450px;">
<h1>HOG Glasses</h1>
<p>How do computers see the world? Upload a photo, and we'll
show you a visualization of how a computer might see it.</p>
<form action="/process" method="post" enctype="multipart/form-data">
<input type="file" name="data">
<input type="submit" value="Process">
</form>
<p>This demo is part of a research project to visualize how computers see the world. <a href="http://mit.edu/vondrick/ihog">Learn more &raquo;</a></p>
</body>
</html>"""
    return resp

@route('/process', method='POST')
def process():
    buffersize = 1024 * 1024
    maxfilesize = 1024 * 1024 * 100
    data = request.files.data
    if data and data.file:
        id = random.randint(0, 999999999999)
        print id
        with open("/scratch/hallucination-daemon/staging/{0}".format(id), "w") as f:
            buffer = data.file.read(buffersize)
            f.write(buffer)
            bytesread = buffersize
            while buffer != "":
                buffer = data.file.read(buffersize)
                f.write(buffer)
                bytesread += buffersize
                if bytesread > maxfilesize:
                    return "File is too big."
        try:
            image = Image.open("/scratch/hallucination-daemon/staging/{0}".format(id))        
        except:
            return "File does not appear to be an image."
        image.convert("RGB").save("/scratch/hallucination-daemon/images/{0}.jpg".format(id))

        while True:
            if os.path.exists("/scratch/hallucination-daemon/out/original-{0}.jpg".format(id)):
                redirect("/show/{0}".format(id))
    else:
        return "You did not upload a file."

@route('/show/<id>')
def show(id):
    resp = """<html><head><title>iHOG Demo</title></head>
<body style="font-family:Arial;">
<div style="margin:20px auto; width:1000px;">
<h1>HOG Glasses</h1>
<p>The left shows the image you uploaded. The right shows how a computer sees the same photo. Notice how likely shadows are removed, fine details are lost, and noise is added. <a href="/">Upload another image &raquo;</a></p>
<table><tr><th>What You See</th><th>What Computers See</th></tr><tr><td>
<img src="/getimage/original-{0}">
</td>
<td>
<img src="/getimage/ihog-{0}">
</td>
</tr>
</table>
</div>
</body>
</html>""".format(id)
    return resp

@route('/getimage/<id>')
def getimage(id):
    return static_file("{0}.jpg".format(id), root="/scratch/hallucination-daemon/out")

run(host="africa.csail.mit.edu", port=8080, debug=True)
