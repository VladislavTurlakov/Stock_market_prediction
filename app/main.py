from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from .routes import router as prediction_router

# Создаем экземпляр приложения FastAPI
app = FastAPI()

# Подключаем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Инициализируем шаблоны Jinja2
templates = Jinja2Templates(directory="templates")

# Подключаем роутер (FastAPI)
app.include_router(prediction_router, prefix="/api", tags=["predictions"])

# Главная страница
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Запуск приложения через ASGI-сервер
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
