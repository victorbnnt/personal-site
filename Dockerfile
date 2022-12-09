FROM nginx:latest
COPY ./web/index.html /usr/share/nginx/html/index.html
COPY .htpasswd /usr/local/.htpasswd
COPY .htaccess /usr/share/nginx/html/.htaccess
#COPY nginx.conf /etc/nginx/

#EXPOSE 80 8080

#CMD ["nginx", "-g", "daemon off;"]
