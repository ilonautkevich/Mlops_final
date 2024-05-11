from streamlit.testing.v1 import AppTest

at = AppTest.from_file("ml_HW2_Alex_R.py")

def test_load_image():
    at = AppTest.from_file("ml_HW2_Alex_R.py")
    at.run(timeout=10) 
  #  at.file_uploader[0].input(open('test_img.png', 'rb')).run()
  #  assert not at.exception

#def test_file_uploader_exists():
    
   # at = AppTest.from_file("ml_HW2_Alex_R.py")
   # at.run()
   # if uploaded_file is not None:
   #   uploaded_file.seek(0)
   #   dico = json.load(uploaded_file)
    #else:
     # dico = json.load(default_file)
    #assert len(at.file_uploader) == 1
    
#def test_load_image_reads_file_data():
 #   at = AppTest.from_file("ml_HW2_Alex_R.py")
    # Simulate file upload
  #  with open("test_img.jpg", "rb") as file:
   #     at.file_uploader[0].input(file).run()
    # Check the returned data
    #if uploaded_file is not None:
     # uploaded_file.seek(0)
      #dico = json.load(uploaded_file)
    #else:
     # dico = json.load(default_file)
    #assert at.return_value == file.read()



   
    

