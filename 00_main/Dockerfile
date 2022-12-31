FROM nginx:latest

RUN ["mv", "/etc/nginx/conf.d/default.conf", "/etc/nginx/conf.d/BACKUP_default.conf"]
#
COPY website /usr/share/nginx/html/.
COPY custom_conf.conf /etc/nginx/conf.d/custom_conf.conf
