"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain, LLMChain
from langchain.llms import OpenAI, OpenAIChat
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate

template = """
Você deve responder as perguntas como Gonçalves Dias, o poeta brasileiro do século XIX.

Lembre-se que você nasceu em Caxias, Maranhão, em agosto de 1823.

Em 1864, você estava muito doente em um navio. O navio se chocou com um banco de areia e naufragou.
Infelizmente, você foi esquecido pelo resgate e acabou morrendo.

Responda em português brasileiro, considerando o contexto da época.

{history}
Você: {human_input}
Gonçalves Dias:
"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template,
)

chain = LLMChain(
        llm=OpenAIChat(model_name="gpt-4", temperature=0), 
        prompt=prompt, 
        verbose=True, 
        memory=ConversationBufferWindowMemory(k=2),
    )

# From here down is all the StreamLit UI.
st.set_page_config(page_title="AGD Experimento N.1.1", page_icon=":book:")
st.header("AGD Experimento N.1.1")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Você: ", "Olá Gonçalves Dias, como vai você?", key="input")
    return input_text


user_input = get_text()

if user_input:
    print(user_input)
    output = chain.run(human_input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style="initials", seed="Gonçalves Dias")
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="initials", seed="vc")
