def completion(client, model:str, prompt:dict)->str:
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response

def chat_completion(chat, model:str, prompt:str)->str:
    response = chat.send_message(str(prompt))
    return response

def prompt_struct(role:str, message:str)->dict:
    return {'role':role, 'message':message}

def chat_append(hist, role,  message):
    hist.append({'role':role, 'message':message})