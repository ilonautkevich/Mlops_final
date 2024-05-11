from streamlit.testing.v1 import AppTest

at = AppTest.from_file("main.py")

def test_load_image():
    at = AppTest.from_file("main.py")
    at.run(timeout=10) 




   
    

