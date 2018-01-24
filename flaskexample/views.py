from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from flask import render_template, request,redirect, url_for, send_from_directory
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from werkzeug.utils import secure_filename
# add the includes

import facebook
import requests
import urllib3
import argparse
import sys
import time

import numpy as np
import tensorflow as tf


# TODO - update to fb calls or read from db
style_to_info = {
  'persian miniature' :
     {'museum_name' : 'Philadelphia Museum of Art',
      'website' : 'http://www.philamuseum.org/collections/permanent/63228.html',
      'fb_events' : []},
  'Japanese Kano School' :
     {'museum_name' : 'Pulitzer Arts Foundation: Sumi-e Painting Workshop Presented by Japan House',
      'website' : 'https://pulitzerarts.org/program/sumi-e-painting-workshop-presented-by-japan-house',
      'fb_events' : ['https://www.facebook.com/events/877780825716060/']},
  'abstract expressionism' :
     {'museum_name' : 'Guggenheim Museum',
      'website' : 'https://www.guggenheim.org/artwork/movement/abstract-expressionism',
      'fb_events' : ['https://www.facebook.com/events/1775280619440369/']},
  'Neo-expressionism' :
     {'museum_name' : 'Guggenheim Museum',
      'website' : 'https://www.guggenheim.org/artwork/movement/neo-expressionism',
      'fb_events' : []},
  'fauvism' :
     {'museum_name' : 'MoMA',
      'website' : 'https://www.moma.org/learn/moma_learning/themes/fauvism',
      'fb_events' : []},
  'cubism paintings' :
     {'museum_name' : 'The Met',
      'website' : 'https://www.metmuseum.org/exhibitions/listings/2014/cubism-leonard-a-lauder-collection',
      'fb_events' : ['https://www.facebook.com/events/110512576391247/']}}
# user = 'amineh' #add your username here (same as previous postgreSQL)
# host = 'localhost'
# dbname = 'birth_db'
# db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_PREFIX = os.path.dirname(os.path.realpath(__file__)) + '/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def call_model(file_name):
  # TODO: move these to parameters
  file_path = UPLOAD_PREFIX + app.config['UPLOAD_FOLDER'] + file_name
  model_file = UPLOAD_PREFIX + "static/models/retrained_graph.pb"
  label_file = UPLOAD_PREFIX + "static/models/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  # End of TODO

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_path,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

  result = []
  for i in top_k:
    result.append({'label': labels[i],
                   'score': results[i],
                   'more_info' : (style_to_info[labels[i]] if labels[i] in style_to_info  else None)})
  return result

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'paintingfile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['paintingfile']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # TO DO: refactor properly before AWS for sure
            print("path is:", UPLOAD_PREFIX + app.config['UPLOAD_FOLDER'] + filename)
            file.save(UPLOAD_PREFIX + app.config['UPLOAD_FOLDER'] + filename)
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('upload-page.html')

@app.route('/about')
def about_us():
  return render_template('about-us.html')

def get_fb_token():
  return 'EAAImEswoEjEBABqwxQZCSggIZCo62S86xKkpLDkaMMvCZBQ1Vex7qR4wwo4BpZB0CcjZBVO4OjM4yRyvfk5Ey8N2ItipSmMNVOhVYAetGA4EFTPVASZAcXW8RZBtncuaFqZA0BxA6ZCaOycNS7lXu9g2oT8PEdaNOvkInyT8ROUZCd6nN7kgytNsESvIXOtOqfJR2f5V3ENy6k7gZDZD'

# mikhaim begim age kasi zad address e amazon ya hamin localhost/eventlist biad inja
# inke che joori miad be in, mohem nist, yani nadooni ham hich tasiri nadare, bedooni ham hamintor
# handles http://localhost/eventlist
# bring up your server
@app.route('/eventlist')
def retrieve_event_list():
  # reading request parameters (i.e. the ones that come after ?blah=foo&blah2=foo1)
  #http://flask.pocoo.org/docs/0.12/quickstart/
  query = request.args.get('query', '')
  graph = facebook.GraphAPI(access_token=get_fb_token(), version = 2.7)
  events = graph.request('/search?q=' + query + '&type=event&limit=100')
  eventList = events['data']
  results = {}
  for i in eventList:
    country=i['place']['location']['country'] if 'place' in i and 'location' in i['place'] and 'country' in i['place']['location'] else None
    if country=='United States':

        # result={}
        event_id=(i['id'])
        # result[event_id]=[i['name'],i['place']['location']['city'],'https://www.facebook.com/events/'+event_id+'/']
        results[event_id] = {
            'name' : i['name'],
            'city' : i['place']['location']['city'],
            'link' : 'https://www.facebook.com/events/'+event_id+'/'
        }
  # return the list to an html page which is under templates folder
  # this says render/process eventlist.html file, and I give you a variable name events
  # which has 'results' now let's create eventlist.html
  return render_template('eventlist.html', events=results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    result = call_model(filename)
    return render_template('results.html', classification_results=result, filename=filename, wikipedia_link='', website_link='')




