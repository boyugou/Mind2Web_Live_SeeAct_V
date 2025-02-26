#Webcanvas evaluate
import re
from urllib.parse import parse_qs, urlparse, unquote
from bs4 import BeautifulSoup
import pdb
import requests
from lxml import html
from .openai import *
from .prompt_constructor import SemanticMatchPromptConstructor


MapTagNameList = [
    "span",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "div",
    "li",
    "ul",
    "p"
]


class StepEvaluator():
    def __init__(self):
        pass


class URLEvaluator(StepEvaluator):

    '''URL Evaluation Scoring'''
    @ staticmethod
    def url_exact_match(input_url, reference_answer, key=False):
        if key:
            try:
                parsed_url = urlparse(input_url)
                url_params = parse_qs(parsed_url.query)
                input_answer = url_params[key][0]
            except:
                return 0
        else:
            input_answer = input_url
        input_answer = unquote(input_answer)
        result_score = MatchFunction.exact_match(
            input_answer, reference_answer)
        # if result_score == 1:
        #     print("url_exactly_match:", input_answer)
        return result_score

    @ staticmethod
    def url_include_match(input_url, reference_answer, key=None):
        # print(input_url, reference_answer)
        if key:
            try:
                parsed_url = urlparse(input_url)
                url_params = parse_qs(parsed_url.query)
                input_answer = url_params[key][0]
            except:
                return 0
        else:
            try:
                parsed_url = urlparse(input_url)
                input_answer = parsed_url.netloc + parsed_url.path
                if parsed_url.fragment is not None and (parsed_url.fragment):
                    input_answer += "#" + parsed_url.fragment
            except:
                input_answer = input_url
        input_answer = unquote(input_answer)
        result_score = MatchFunction.include_match(
            input_answer, reference_answer)
        # print("score:", result_score, input_answer)
        return result_score

    @ staticmethod
    def url_semantic_match(input_url, semantic_method, key=False):
        if key:
            try:
                parsed_url = urlparse(input_url)
                url_params = parse_qs(parsed_url.query)
                input_answer = url_params[key][0]
            except:
                return 0
        else:
            input_answer = input_url
        input_answer = unquote(input_answer)
        result_score = MatchFunction.semantic_match(input_answer, semantic_method)
        return result_score


class ElementEvaluator(StepEvaluator):
    '''Element evaluation and scoring'''

    @staticmethod
    def is_same_element(page, input_coord, reference_element_handle):
        x,y=input_coord
        # Get the bounding box of the element
        bounding_box = reference_element_handle.bounding_box(timeout=2000)

        if bounding_box:
            # Extract the bounding box coordinates
            element_x = bounding_box['x']
            element_y = bounding_box['y']
            element_width = bounding_box['width']
            element_height = bounding_box['height']
            # Check if the given (x, y) is within the bounding box
            if (element_x <= x <= element_x + element_width and
                element_y <= y <= element_y + element_height):
                #print("Examined Bounding box",bounding_box,"input answer coord",input_coord,'which is True',flush=True)
                return True
        #print("Examined Bounding box",bounding_box,"input answer coord",input_coord,'which is False',flush=True)
        return False

    @ staticmethod
    def path_exact_match(input_answer, reference_answer, method, page, input_netloc, reference_netloc):
        score = 0
        if method == "xpath":
            if reference_netloc != input_netloc:
                # print("reference_netloc:", reference_netloc,
                #       "input_netloc:", input_netloc)
                return 0
            try:
                html_content = page.content()
                tree = html.fromstring(html_content)
                input_elements = tree.xpath(input_answer)
                reference_elements = tree.xpath(reference_answer)
            except:
                score = 0
            if (input_elements is not None) and (reference_elements is not None):
                score = input_elements[0] is reference_elements[0]
                try:
                    if reference_elements[0].tag in MapTagNameList:
                        trace_up_count = 0
                        current_element = reference_elements[0]
                        while trace_up_count < 3 and score == 0:
                            trace_up_count += 1
                            current_element = current_element.getparent()
                            parent_score = input_elements[0] is current_element
                            score = max(score, parent_score)
                except:
                    pass
            else:
                score = 0
        elif method == "selector":
            if reference_netloc != input_netloc:
                return 0

            try:
                input_element = input_answer#input element is input coord
                reference_element = page.locator(reference_answer)
                #input_element_handle = input_element.element_handle()
                #reference_element_handle = reference_element.element_handle()
                if (input_element is not None) and (reference_element is not None):
                    score = ElementEvaluator().is_same_element(page, input_coord=input_element, reference_element_handle=reference_element)
                    '''
                    try:
                        reference_tag = page.evaluate("(element) => element.tagName.toLowerCase()", reference_element_handle)
                        if reference_tag in MapTagNameList:
                            trace_up_count = 0
                            current_element = reference_element
                            while trace_up_count < 3 and score == 0:
                                trace_up_count += 1
                                parent_element = current_element.locator("xpath=..")
                                parent_element_handle = parent_element.element_handle()
                                current_element = parent_element
                                if parent_element:
                                    parent_score = ElementEvaluator().is_same_element(page, input_element_handle=input_element_handle, reference_element_handle=parent_element_handle)
                                    score = max(score, parent_score)
                    except Exception as e:
                        print(e)
                        pass
                    '''
            except:
                score = 0
        # result_score = MatchFunction.include_match(
        #     input_answer, reference_answer)
        return score

    @ staticmethod
    def path_included_match(input_answer, reference_answer, method, html_content):
        # TODO Add path inclusion matching method
        result_score = MatchFunction.include_match(
            input_answer, reference_answer)
        return result_score

    @ staticmethod
    def element_value_exact_match(input_answer, reference_answer, input_netloc, reference_netloc):
        if reference_netloc != input_netloc:
            # print("reference_netloc:", reference_netloc,
            #       "input_netloc:", input_netloc)
            return 0
        result_score = MatchFunction.exact_match(
            input_answer, reference_answer)
        return result_score

    @ staticmethod
    def element_value_include_match(input_answer, reference_answer, input_netloc, reference_netloc):
        if reference_netloc != input_netloc:
            # print("reference_netloc:", reference_netloc,
            #       "input_netloc:", input_netloc)
            return 0
        result_score = MatchFunction.include_match(
            input_answer, reference_answer)
        return result_score

    @ staticmethod
    def element_value_semantic_match(input_answer, semantic_method, input_netloc, reference_netloc=0):
        if reference_netloc != input_netloc:
            # print("reference_netloc:", reference_netloc,
            #       "input_netloc:", input_netloc)
            return 0
        if len(input_answer) == 0:
            return 0
        result_score = MatchFunction.semantic_match(input_answer, semantic_method)
        return result_score


class TextEvaluator(StepEvaluator):
    '''Text evaluation and scoring'''
    @ staticmethod
    def text_exact_match(input_answer, reference_answer):
        result_score = MatchFunction.exact_match(
            input_answer, reference_answer)
        return result_score

    @ staticmethod
    def text_included_match(input_answer, reference_answer):
        result_score = MatchFunction.include_match(
            input_answer, reference_answer)
        return result_score

    @ staticmethod
    def text_semantic_match(input_answer, semantic_method):
        result_score = MatchFunction.semantic_match(
            input_answer, semantic_method, semantic_method)
        return result_score


class MatchFunction():
    def __init__(self):
        pass

    @ staticmethod
    def exact_match(input_answer, reference_answer) -> int:
        return 1 if input_answer == reference_answer else 0

    @ staticmethod
    def include_match(input_answer, reference_answer) -> int:
        return 1 if reference_answer in input_answer else 0

    @ staticmethod
    def semantic_match(input_answer, semantic_method) -> float:

        GPT4 = GPTGenerator4()
        semantic_request = SemanticMatchPromptConstructor(
        ).construct(input_answer, semantic_method)
        score = None
        for i in range(3):
            try:
                response = GPT4.generate(semantic_request)
                score = re.findall("```(.*?)```", response, re.S)[0]
                score = eval(score)
                # Limit the score between 0 and 1
                score = max(0, min(1, score))
                if score != None:
                    break
            except:
                score = None
        if score == None:
            score = 0
        if score != 0 and score != 1:
            return round(score, 2)
        else:
            return score
