from bottle import route, run, get, post, request, static_file
import time, os
import tensorflow as tf, sys
 
graph_path = 'output_graph.pb'
labels_path = 'output_labels.txt'
minimum_score = 0.90

@route('/hello')
def hello():
    return "Hello World!"


@get('/') # or @route('/login')
def login():
    return '''
        <center>
        <br><br><br><br><br>
        <form action="/upload" method="post" enctype="multipart/form-data">
            Image: <input name="image" type="file" />
            <input value="Submit" type="submit" />
        </form>
    '''


@post('/upload') # or @route('/login', method='POST')
def do_upload():
    image = request.files.get('image')

    name, ext = os.path.splitext(image.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'
    new_name = str(time.time())+""+ext
    save_path = "uploads/"+new_name
    image.save(save_path) # appends upload.filename automatically
    
    scores = predict(new_name)

    list_scores = "<b>RESULTS</b><br>"
    top_predict = ""
    for res in scores:

        if res['score'] >= minimum_score:
            top_predict = res['label']

        list_scores += res['label']+" = "+str(res['score'])+"<br>"

    if len(top_predict) > 0 :
        top_predict = "<h2 style='color:green'>INI ADALAH GAMBAR BUNGA "+top_predict.upper()+"</h2>"
    else:
        top_predict = "<h2 style='color:red'>INI BUKAN GAMBAR BUNGA </h2>"

    return "<body style='font-size: 35px;'><br><br><center><p><img src='/static/{}' width='300px'></p>{}<p>{}</p></body>".format(new_name,top_predict,list_scores)    


@route('/static/<filename>')
def server_static(filename):
    return static_file(filename, root='./uploads')


def predict(filename):

    image_path = 'uploads/'+filename

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
        in tf.gfile.GFile(labels_path)]
    
    # Unpersists graph from file
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    data_all = []
    # Feed the image_data as input to the graph and get first prediction
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, 
        {'DecodeJpeg/contents:0': image_data})
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

            response = {}
            response['label'] = human_string
            response['score'] = score

            data_all.append(response)
            
    
    return data_all


run(host='localhost', port=1111, debug=True)

