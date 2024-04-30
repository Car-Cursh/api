from datetime import datetime, timedelta, timezone
from typing import Annotated, Union

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Request
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

from pydantic import BaseModel

import uvicorn

import numpy as np
import cv2
from io import BytesIO

SECRET_KEY = "6b3281e4c99f069c29a22f59cf39b91628f832970662133af95aac47739cc0a1"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
print(pwd_context.hash("jo"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token",auto_error=False)

app = FastAPI()

fake_users_db = {
    "root" : {
        "username" : "root",
        "full_name" : "Admin_root",
        "email" : "admin@carcrush.com",
        "hashed_pw" : "$2b$12$Z1CQg65HTlBUV0Yh8rpkAeics1Zg6rVRx33Cxz0q4/POKB6tWHJwG",
        "disabled" : False
    }
}

class Token(BaseModel):
    access_token : str
    token_type : str

class TokenData(BaseModel):
    username: Union[str,None] = None

class User(BaseModel):
    username : str
    email : Union[str,None] = None
    full_name : Union[str,None] = None
    disabled : Union[bool,None] = None

class UserInDB(User):
    hashed_pw : str

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    print(exc.name)
    return FileResponse("view/login.html")

def verify_password(plain_pw, hashed):
    return pwd_context.verify(plain_pw, hashed)

def get_pw_hash(pw):
    return pwd_context.hash(pw)

def get_user(db, username : str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username : str, pw : str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(pw,user.hashed_pw):
        return False
    return user

def create_access_token(data: dict, expires_delta : Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp" : expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_token_from_cookie(request : Request):
    token = request.cookies.get("access_token")
    return token

async def get_current_user(token : Annotated[str, Depends(get_token_from_cookie)]):
    print("BOR")
    # cr_exception = HTTPException(
    #     status_code = status.HTTP_401_UNAUTHORIZED,
    #     detail = "Can't validate credentials",
    #     headers = {"WWW-Authenticate":"Bearer"}
    # )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        print(username)
        if username is None:
            raise UnicornException(name="your payload userame is weirdo")
        token_data = TokenData(username=username)
    except:
        raise UnicornException(name="non-login, except")
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise UnicornException(name="your not in user db")
    return user

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    print("all i want is yuo")
    return current_user





@app.post("/token")
async def login_for_access_token(
    form_data : Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    response = RedirectResponse("/", status.HTTP_302_FOUND)
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True
    )
    return response

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.get("/")
async def main(current_user : Annotated[User, Depends(get_current_active_user)]):
    if not current_user:
        return FileResponse('view/index.html')
    else:
        return FileResponse('view/index.html')

@app.post("/upload")
async def create_upload_file(file: UploadFile, current_user : Annotated[User, Depends(get_current_active_user)] ):
    if not current_user:
        return FileResponse('view/login.html')
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert back to bytes
    _, encoded_img = cv2.imencode('.PNG', gray)
    byte_io = BytesIO(encoded_img.tobytes())

    return StreamingResponse(BytesIO(encoded_img.tobytes()), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)