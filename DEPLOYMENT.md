# 🚀 Smart Harvest AI - Deployment Guide

## Quick Deploy Options

### Option 1: Render (Recommended - Free Tier Available)

1. **Create account** at [render.com](https://render.com)

2. **Connect GitHub**
   - Fork/push this repo to your GitHub
   - Connect your GitHub account to Render

3. **Create Web Service**
   - Click "New +" → "Web Service"
   - Connect your repository
   - Configure:
     - **Name**: `smart-harvest-ai`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn --chdir backend app_v2:app --bind 0.0.0.0:$PORT`
     - **Instance Type**: Free

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Your app will be live at: `https://smart-harvest-ai.onrender.com`

---

### Option 2: Railway (Easy & Fast)

1. **Visit** [railway.app](https://railway.app)

2. **Deploy from GitHub**
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure**
   - Railway auto-detects Python
   - Add environment variable:
     - `PORT`: `5001`

4. **Deploy**
   - Automatic deployment starts
   - Get your URL from dashboard

---

### Option 3: Heroku (Classic Option)

1. **Install Heroku CLI**
   ```bash
   brew install heroku/brew/heroku  # macOS
   ```

2. **Login & Create App**
   ```bash
   heroku login
   heroku create smart-harvest-ai
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

4. **Open App**
   ```bash
   heroku open
   ```

---

### Option 4: PythonAnywhere (Simple Hosting)

1. **Create account** at [pythonanywhere.com](https://www.pythonanywhere.com)

2. **Upload code**
   - Use "Files" tab to upload project
   - Or clone from GitHub

3. **Create Web App**
   - Go to "Web" tab
   - Click "Add a new web app"
   - Choose Flask
   - Point to `backend/app_v2.py`

4. **Configure WSGI**
   - Edit WSGI configuration file
   - Point to your Flask app

---

### Option 5: AWS EC2 (Production Grade)

1. **Launch EC2 Instance**
   - Ubuntu 22.04 LTS
   - t2.micro (free tier)

2. **SSH & Setup**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Install dependencies
   sudo apt update
   sudo apt install python3-pip nginx
   
   # Clone repo
   git clone https://github.com/yourusername/smart_harvest.ai
   cd smart_harvest.ai
   
   # Install requirements
   pip3 install -r requirements.txt
   ```

3. **Run with Gunicorn**
   ```bash
   gunicorn --chdir backend app_v2:app --bind 0.0.0.0:5001 --workers 4
   ```

4. **Setup Nginx (Optional)**
   - Configure reverse proxy
   - Add SSL with Let's Encrypt

---

## Environment Variables

Set these in your hosting platform:

```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
PORT=5001  # Or platform default
```

---

## Pre-Deployment Checklist

- [ ] All model files in `backend/models/` directory
- [ ] `requirements.txt` is up to date
- [ ] Secret key is changed from default
- [ ] CORS settings configured for your domain
- [ ] Test locally with `gunicorn --chdir backend app_v2:app`

---

## Post-Deployment

1. **Test the deployment**
   ```bash
   curl https://your-app-url.com/api/health
   ```

2. **Monitor logs**
   - Check platform dashboard for errors
   - Monitor memory usage (models are ~100MB)

3. **Custom Domain (Optional)**
   - Add your domain in platform settings
   - Update DNS records
   - Enable HTTPS

---

## Troubleshooting

### Models not loading
- Ensure model files are committed to git
- Check file paths are correct
- Verify memory limits (need ~512MB minimum)

### Geolocation not working
- Requires HTTPS in production
- Most platforms provide free SSL

### Slow cold starts
- Use paid tier for always-on instances
- Or implement model caching

---

## Cost Estimates

| Platform | Free Tier | Paid (Basic) |
|----------|-----------|--------------|
| Render | 750 hrs/month | $7/month |
| Railway | $5 credit | $5/month |
| Heroku | Eco $5/month | $7/month |
| PythonAnywhere | Limited | $5/month |
| AWS EC2 | 750 hrs/month | $10/month |

---

## Support

For issues, check:
- Platform status pages
- Application logs
- GitHub Issues

**Your app is ready to deploy! Choose any option above and go live in minutes.** 🚀
