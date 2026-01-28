from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import tiktoken
import os