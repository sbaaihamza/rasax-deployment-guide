# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
 
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk import FormValidationAction
from rasa_sdk.types import DomainDict

import random
from datetime import datetime
import pandas as pd
import json
from functions import *

# load static data from file
js_pth='/app/actions/mydata/static_data.json'
definitions_dict, module_titles,choose_qst_variations,definitions_dict_old,other_qst_variations= load_data_from_file(js_pth)

class ActionStopNavigation(Action):
    def name(self) -> str:
        return "action_stop_navigation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        dispatcher.utter_message("بالتأكيد! إذا كان لديك أي أسئلة أخرى أو إذا كنت بحاجة إلى مزيد من المعلومات، فلا تتردد في طرحها. أنا هنا للمساعدة")
        return []
    
class ActionGetUserQuestion(Action):
    def name(self) -> str:
        return "action_get_user_question"
    ##new
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain) -> list:
        user_message_all = tracker.latest_message.get('text')
        # Generate recommendations
        try:
            df_recommendations = provide_recommendations(user_message_all, THRESH=0.3, n=1000, unique_values_dict=unique_values_dict, BERT_weights=BERT_weights,n_module=n_module)
            dataframe_json = df_recommendations.to_json(orient='split')
        except Exception as e:
            print(e)
            dispatcher.utter_message("!! الرجاء المحاولة مرة أخرى") 
            return [SlotSet("user_question", user_message_all)]
        # Handle the case where no recommendations are found
        if df_recommendations.empty:
            dispatcher.utter_message("من فضلك أعد صياغة سؤالك")
            return [SlotSet("user_question", user_message_all)]
        # Generate and send module buttons
        module_ids, module_names = module_recommendations(df_recommendations, n=n_module)
        button_list = [{"title": name, "payload": f'/inform_module{{"module_id":"{str(module_id)}"}}'} for module_id, name in zip(module_ids, module_names)]
        dispatcher.utter_message(text="اختر الوحدة المتعلقة بسؤالك", buttons=button_list)
        # Set the user_question value in a slot for future use
        return [SlotSet("user_question", user_message_all) , SlotSet("my_dataframe_slot", dataframe_json)]   


class ActionReselectModule(Action):
    def name(self) -> str:
        return "action_reselect_module"

    ##new
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain) -> list:
        # Try reading the user track from Excel file
        try:
            my_dataframe_slot = tracker.get_slot('my_dataframe_slot')
            df_recommendations = pd.read_json(my_dataframe_slot, orient='split')            
        except Exception as e:
            print(e)
            dispatcher.utter_message(" !! خلل في تحميل البيانات")
            return []
        # Handle the case where no resp are found
        if df_recommendations.empty:
            dispatcher.utter_message("من فضلك أعد صياغة سؤالك")
            return []
        # Generate and send module buttons
        module_ids, module_names = module_recommendations(df_recommendations, n=n_module)
        button_list = [{"title": name, "payload": f'/inform_module{{"module_id":"{module_id}"}}'} 
                    for module_id, name in zip(module_ids, module_names)]
        dispatcher.utter_message(text="اختر الوحدة المتعلقة بسؤالك", buttons=button_list)
        return []
    
class ActionGetModuleId(Action):
    def name(self):
        return "action_get_module_id"

    def run(self, dispatcher, tracker, domain):
        # Extract payload from the latest message

        latest_entities = tracker.latest_message.get("entities", [])

        if latest_entities:
    # Assuming you have only one entity in the latest message
            entity_name = latest_entities[0].get("entity")
            entity_value = latest_entities[0].get("value")
            return [SlotSet(entity_name, entity_value)]
        else:
            return []


class ActionGet_Situations(Action):
    def name(self):
        return "action_get_situations"

    def run(self, dispatcher, tracker, domain):
        # Access the ID from the slot
        module_number = tracker.get_slot('module_id')
        try:
            my_dataframe_slot = tracker.get_slot('my_dataframe_slot')
            df_rslt = pd.read_json(my_dataframe_slot, orient='split')

        except Exception as e:
            print(e)
            dispatcher.utter_message(" !! خلل في تحميل البيانات")#TODO: translate
            return []
        situation_ids,situation_names=situation_recommendations(df_rslt,int(module_number),n=n_situation)
        if situation_ids==[]:
                        dispatcher.utter_message("لا يوجد السياق متاح في هذه الوحدة")
        else:

            button_list = [{"title": situation_names[i], "payload": f'/inform_situation{{"situation_id":"{str(situation_ids[i])}"}}'  } for i in range(len(situation_ids))]
            button_list.append({"title": "انقر هنا لإعادة إختيار الوحدة", "payload": '/rechoisir_module'})
            dispatcher.utter_message(text= "اختر السياق الأقرب إلى سؤالك",buttons=button_list)
        return []

class ActionGetsituationId(Action):
    def name(self):
        return "action_get_situation_id"

    def run(self, dispatcher, tracker, domain):
        # Extract payload from the latest message
        latest_entities = tracker.latest_message.get("entities", [])

        if latest_entities:
    # Assuming you have only one entity in the latest message
            entity_name = latest_entities[0].get("entity")
            entity_value = latest_entities[0].get("value")
            return [SlotSet(entity_name, entity_value)]
        else:
            return []


class ActionGet_Questions(Action):
    def name(self):
        return "action_get_questions"

    def run(self, dispatcher, tracker, domain):
        # Access the ID from the slot
        situation_number = tracker.get_slot('situation_id')
        try:
            my_dataframe_slot = tracker.get_slot('my_dataframe_slot')
            df_rslt = pd.read_json(my_dataframe_slot, orient='split')
        except Exception as e:
            print(e)
            dispatcher.utter_message(" !! خلل في تحميل البيانات")
            return []
        question_ids,question_names,reste,reste_question=question_recommendations(df_rslt,int(situation_number),n=n_question)
        if question_ids==[]:
                        dispatcher.utter_message("لا يوجد سؤال متاح في هذا السياق")
        else:

            random.shuffle(choose_qst_variations)
            button_list = [{"title": question_names[i], "payload": f'/inform_question{{"question_id":"{str(question_ids[i])}"}}' } for i in range(len(question_ids))]
            dispatcher.utter_message(text= choose_qst_variations[0],buttons=button_list)
        return[]  
    
class ActionGetQuestionId(Action):
    def name(self):
        return "action_get_question_id"

    def run(self, dispatcher, tracker, domain):
        # Extract payload from the latest message
        latest_entities = tracker.latest_message.get("entities", [])

        if latest_entities:
    # Assuming you have only one entity in the latest message
            entity_name = latest_entities[0].get("entity")
            entity_value = latest_entities[0].get("value")
            return [SlotSet(entity_name, entity_value)]
        else:
            return []
    
class ActionGet_Response(Action):
    def name(self):
        return "action_get_response"

    def run(self, dispatcher, tracker, domain):
        # Access the ID from the slot
        question_number = tracker.get_slot('question_id')
        response=get_responses(int(question_number))
        # Use the ID in your action logic
        dispatcher.utter_message(text=f" {response[0]}")

        # Shuffle the messages randomly
        random.shuffle(other_qst_variations)

        # Choose and send one of the messages
        response = other_qst_variations[0]
        dispatcher.utter_message(text=response)

        return []
    

class ActionUtterModuleButtons(Action):
    def name(self) -> Text:
        return "action_utter_module_buttons"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Create buttons based on the list of module titles
        button_list = [{"title": title, "payload": f"module_definitions{i+1}"} for i, title in enumerate(module_titles)]
        dispatcher.utter_message(text= "اضغط على اسم الوحدة للحصول على تعريفها", buttons=button_list)

        return []
    
    
class ActionGetModuleDefinitions2(Action):
    def name(self) -> Text:
        return "action_get_module_definitions2"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get the entire dictionary of definitions
        user_selection = tracker.latest_message.get('text')

        # Send the entire dictionary as a response
        dispatcher.utter_message(text=str(definitions_dict[user_selection]))

        return []

### to remove this action later
class LogConversation(Action):
    def name(self) -> Text:
        return "action_log_conversation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Extract conversation data with message types and sender IDs
        sender_id = tracker.sender_id
        conversation_data = []

        for event in tracker.events:
            if 'text' in event:
                message = event['text']
                message_type = 'user' if event['event'] == "user" else 'bot'
                time = event.get('timestamp', '')
                conversation_data.append({'sender_id': sender_id, 'message': message, 'message_type': message_type, 'Time': time})

        # Format the date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Specify the file path
        file_path = "conversation_log.txt"

        # Open the file in append mode and write the formatted datetime and conversation data
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(f"Timestamp: {current_datetime}\n")
            for entry in conversation_data:
                # file.write(f"(sender_id: {entry['sender_id']}){entry['message_type'].capitalize()}: {entry['message']} (Time: {entry['Time']})\n")
                file.write(f"sender_id: {entry['sender_id']} , {entry['message_type'].capitalize()}: {entry['message']} , Time: {entry['Time']}\n")

            file.write("\n")  # Add a newline between conversations

        return []
    

    
