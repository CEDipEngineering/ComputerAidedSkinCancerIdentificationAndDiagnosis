import {api} from "../api"

export async function uploadImage(base64){
    return api.post("upload", {"imagedata": base64}).then(
        response => {return response}
    )
}