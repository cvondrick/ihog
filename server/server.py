from bottle import route, run, request, static_file, redirect
import random
import Image
import os

@route('/')
@route('/index')
def index():
    resp = """<html><head><title>iHOG Demo</title></head>
<body>
<form action="/process" method="post" enctype="multipart/form-data">
<input type="file" name="data">
<input type="submit" value="Process">
</form>
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
        image.save("/scratch/hallucination-daemon/images/{0}.jpg".format(id))

        while True:
            if os.path.exists("/scratch/hallucination-daemon/out/{0}.jpg".format(id)):
                redirect("/show/{0}".format(id))
    else:
        return "You did not upload a file."

@route('/show/<id>')
def show(id):
    resp = """<html><head><title>iHOG Demo</title></head>
<body>
<img src="/getimage/{0}">
</body>
</html>""".format(id)
    return resp

@route('/getimage/<id>')
def getimage(id):
    return static_file("{0}.jpg".format(id), root="/scratch/hallucination-daemon/out")

run(host="africa.csail.mit.edu", port=8080, debug=True)
