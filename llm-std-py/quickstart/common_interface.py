from langchain_ollama import ChatOllama


model = ChatOllama(
    model="gemma3:12b"
)
print("\n#################### invoke() ##########################\n")
completion = model.invoke('올라마 반가워!')
print(completion)

print("\n################### batch() ##########################\n")
completions = model.batch(['올라마 반가워', '넌 어떤 모델이니?'])
print(completions)
print("\n###################### stream() #########################\n")
for token in model.stream('만나서 반가웠어!'):
    print(token)