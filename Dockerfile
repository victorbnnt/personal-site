FROM nginx:latest
COPY ./web/index.html /usr/share/nginx/html/index.html
COPY .htpasswd /usr/local/.htpasswd
COPY .htaccess /usr/share/nginx/html/.htaccess
