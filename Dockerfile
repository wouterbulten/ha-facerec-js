FROM node

COPY package.json /facerec/package.json
COPY package-lock.json /facerec/package-lock.json
WORKDIR /facerec
RUN npm install

# Copy full source later
COPY . /facerec

CMD npm start