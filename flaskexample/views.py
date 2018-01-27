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

import json
import urllib.parse
import urllib.request
import us

import numpy as np
import tensorflow as tf


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

def find_style(file_name):
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

  #top_k = results.argsort()[-5:][::-1]
  top = results.argsort()[-1]
  labels = load_labels(label_file)
  print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

  return labels[top] if top < len(labels) else None

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
  # app_id = '410719039365960',
  # app_secret = 'd230b52f340fb18d80c6dcdf4dbfb27e'
  # payload = {'grant_type': 'client_credentials', 'client_id': app_id, 'client_secret': app_secret}
  # file = requests.post('https://graph.facebook.com/oauth/access_token?', params = payload)
  # result = json.loads(file.text)['access_token']
  result = 'EAAB7ZAIUemhABADDwiIpQfMb2RLeJP8cwH9yBv05XGUgPma24uoZCZB8iF8m2kaQVEIsFw5zL4P4W8kPosKCgU6C5XZAlZCTvk1Um0irdyhZCJZBIQNqfaZBTn24FZAThqtQI8H6aqXTpfZBosdPBeb2VG0BiHS05oJ3EaXsGE5K1TIgZDZD'
  return result

@app.route('/eventlist')
def find_fb_events(style):
  # reading request parameters (i.e. the ones that come after ?blah=foo&blah2=foo1)
  #http://flask.pocoo.org/docs/0.12/quickstart/
  events = []
  try:
    graph = facebook.GraphAPI(access_token=get_fb_token(), version = 2.7)
    events = graph.request('/search?q=' + style + '&type=event&limit=100')
    eventList = events['data']
    events = {}
    print(len(eventList))
    for i in eventList:
      country=i['place']['location']['country'] if 'place' in i and 'location' in i['place'] and 'country' in i['place']['location'] else None
      # if country=='United States':
      event_id=(i['id'])
      location = i['place']['location']['city'] + ', ' + i['place']['location']['state'] if country=='United States' else country
      events[event_id] = {
          'title' : i['name'],
          'location' : location,
          'link' : 'https://www.facebook.com/events/'+event_id+'/'
      }
  except Exception as e:
    print(e)
  return events


def find_museums(style, result_count=20):
  api_key = 'AIzaSyDQ57s9pTYh8yi3jwMXY_A2ZIzYc27ds9s'
  service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
  params = {
      'query': style,
      'limit': result_count,
      'indent': True,
      'key': api_key,
      'types':'Organization',
      # 'types':Event
  }
  url = service_url + '?' + urllib.parse.urlencode(params)
  response = json.loads(urllib.request.urlopen(url).read())
  state_names = [state.name for state in us.states.STATES_AND_TERRITORIES]
  results=[]
  for element in response['itemListElement']:
      description=element['result']['description'] if 'description' in element['result'] else None
      url=element['result']['url'] if 'url' in element['result'] else None
      name=element['result']['name'] if 'name' in element['result'] else None
      city=element['result']['city'] if 'city' in element['result'] else None
      sec_url=element['result']['detailedDescription']['url'] if 'detailedDescription' in element['result'] and 'url' in element['result']['detailedDescription'] else None
      if url is None:
          url=sec_url
      if description is not None:
          words=description.split()
          state=words[-1]
          if state in state_names:
              results.append({'link': url , 'title': name, 'city' : city })
  return results

@app.route('/museumlist')
def museums_page():
  museums = get_museum_list(request.args.get('query', ''))
  return get_render_template('museumlist.html', events=results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
  style = find_style(filename)
  museums = find_museums(style)
  events = find_fb_events(style)
  return render_template('results.html',fb_events=events, museums= museums, style=style, filename=filename)



