import axios from "axios"

const baseUrl = "http://192.168.15.60:8000/"
export const api = axios.create({baseURL: baseUrl});