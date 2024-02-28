import config
import pyodbc
import jwt
import hashlib


conn = pyodbc.connect('DRIVER='+config.DB_DRIVER+';SERVER='+config.DB_URL+',1433;DATABASE='+config.DB_NAME+';UID='+config.DB_USERNAME+';PWD='+config.DB_PWD)
cursor = conn.cursor()
class User:
    # This class is used to handle user related operations
    def __init__(self):
        return
    
    # This method is used to create a new user
    def create(self, displayName="", email="", password=""):
        user = self.get_by_email(email)

        # check if user already exists
        if user:
            return
        
        hashed_password = self.encrypt_password(password)

        # create new user
        cursor.execute("INSERT INTO users (displayName, email, password) VALUES (?, ?, ?)", displayName, email, hashed_password)
        cursor.execute("INSERT INTO user_role (role) VALUES (?, ?, ?)", user['id'], 'user')
        conn.commit()

        new_user = self.get_by_email(email)
        
        return new_user
    
    # This method is used to login a user
    def login(self, email, password):
        """Login a user"""
        print(email, password)
        user = self.get_by_email(email)

        if not user or not self.check_password(password, user["password"]):
            return
        user.pop("password")
        return user

    # This method is get a user by id
    def get_by_id(self, user_id):
        """Get a user by id"""
        user = cursor.execute("SELECT * FROM users WHERE _id=?", user_id)

        if not user:
            return { "message": "User not found", "data": None, "error": "Not found" }, 404
        user.pop("password")
        return user
    
    # This method is get a user by email
    def get_by_email(self, email):
        """Get a user by email"""
        cursor.execute("SELECT id, email, password, displayName FROM users WHERE email=?", email)

        user = [dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()] or None
        if not user:
            return
        user = user[0]
        return user

    # This method is used to encrypt the password
    def encrypt_password(self, password):
        """Encrypt the password using a hashing algorithm"""
        hash_obj = hashlib.sha256()
        hash_obj.update(password.encode())
        hashed_password = hash_obj.hexdigest()
        return hashed_password
    
    # This method is used to check if the password matches the hashed password
    def check_password(self, password, hashed_password):
        """Check if the password matches the hashed password"""
        return hashlib.sha256(password.encode()).hexdigest() == hashed_password