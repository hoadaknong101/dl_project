import praw
import dotenv

reddit = praw.Reddit(
    client_id = dotenv.get_key('.env','REDDIT_CLIENT_ID'),
    client_secret = dotenv.get_key('.env','REDDIT_CLIENT_SECRET'),
    user_agent="Final project by Pham Dinh Quoc Hoa and Nguyen Phuong Thinh"
)