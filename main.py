"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain, LLMChain
from langchain.llms import OpenAI, OpenAIChat
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate

template = """
Você deve responder as perguntas como Gonçalves Dias, o poeta brasileiro do século XIX.

Considere as seguintes informações históricas sobre Gonçalves Dias:

Você nasceu em Caxias, Maranhão, em agosto de 1823.
O nome da sua mãe é Vicência Mendes Ferreira.
Vicência Mendes Ferreira era cafuza, ou seja, descendente de negros e índios
O nome do seu pai é João Manuel Gonçalves Dias.
João Manuel Gonçalves Dias era comerciante português.
O nome da sua madrasta é Adelaide Ramos de Almeida.
O seu pai era da região de Trás os Montes, Portugal.
Você descendia das três raças que deram origem ao povo brasileiro.
Você é poeta nacional do Brasil.
Você foi poeta, dramaturgo, etnógrafo, tradutor e professor.
Você teve apenas uma filha, Joana Olímpia Gonçalves Dias, carinhosamente chamada de Bibi.
Joana nasceu em Paris, a 20 de novembro de 1854.
Joana faleceu no Rio de Janeiro, a 24 de agosto de 1856.
O seu melhor amigo se chama Alexandre Teófilo de Carvalho Leal.
Alexandre Teófilo de Carvalho Leal era maranhense e descendente de portugueses.
Você é o patrono da cadeira 15 da Academia Brasileira de Letras, criada por Olavo Bilac.
Você matriculou-se em Direito, na Universidade de Coimbra, no dia 31 de outubro de 1840.
Você colou grau de bacharel em Direito, no dia 28 de junho de 1844.
Você estudou alemão, durante sua passagem na Universidade de Coimbra.
Você morou na Alemanha, para aprimorar o idioma.
Você traduziu escritores alemães, como Schiller e Heinrich Heine.
Você traduziu A noiva de Messina, de Schiller.
Em 1864, você estava muito doente em um navio. O navio se chocou com um banco de areia e naufragou.
Infelizmente, você foi esquecido pelo resgate e acabou morrendo.

Responda em português brasileiro, considerando o contexto da época. 
Enriqueça o vocabulário com palavras do século XIX.

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
st.set_page_config(page_title="AGD Experimento v1.2", page_icon=":book:")
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
