import axios from "axios"

const baseUrl = "http://10.102.30.137:8000/"
export const api = axios.create({baseURL: baseUrl});