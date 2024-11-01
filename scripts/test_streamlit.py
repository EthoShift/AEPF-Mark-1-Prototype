import streamlit as st

def main():
    st.title("AEPF Test App")
    st.write("If you can see this, Streamlit is working!")
    
    if st.button("Click me"):
        st.write("Button clicked!")

if __name__ == "__main__":
    main() 