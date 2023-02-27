def get_list():
  with open("순서.txt", "r") as f:
      example = f.readlines()
  _dict={}

  for i in range(len(example)):
    a=example[i].replace('\n','')
    _dict[i]=a
  return _dict
