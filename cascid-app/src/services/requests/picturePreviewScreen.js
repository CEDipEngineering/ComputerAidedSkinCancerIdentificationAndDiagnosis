import {api} from "../api"

export async function uploadImage(base64){
    return api.post("upload", {image_to_base64: base64}).then(
        response => {return response}
    )
}

export async function predictImage(uuid){
    return api.get(`images/${uuid}`).then(
        response => {return response}
    )
}